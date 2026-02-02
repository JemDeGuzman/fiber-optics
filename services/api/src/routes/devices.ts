import express, { Request, Response } from "express";
import axios from "axios";
import { getLocalIp } from "../utils/ipDetection";

const router = express.Router();

interface ConnectedDevice {
  deviceId: string;
  deviceIp: string;
  connectedAt: Date;
  lastHeartbeat: Date;
  receiveUrl?: string;
  clientType?: string;
  pendingMessages?: any[];
}

// Store connected devices in memory (in production, use a database)
const connectedDevices: Map<string, ConnectedDevice> = new Map();

/**
 * Register a remote device's IP locally on this server
 * POST /api/devices/register
 * Body: { remoteIp: string }
 * Returns: { success: boolean, thisDeviceIp: string, message: string }
 */
router.post("/register", (req: Request, res: Response) => {
  try {
    const { remoteIp } = req.body;

    if (!remoteIp) {
      return res.status(400).json({
        success: false,
        message: "Missing remoteIp",
      });
    }

    // Generate a unique device ID for this registration
    const deviceId = `remote-${remoteIp}-${Date.now()}`;

    // Store the remote device locally on this server
    connectedDevices.set(deviceId, {
      deviceId,
      deviceIp: remoteIp,
      connectedAt: new Date(),
      lastHeartbeat: new Date(),
      pendingMessages: [],
    });

    // Return this server's IP (for the remote device to save, if needed)
    const thisDeviceIp = getLocalIp();

    console.log(`✓ Remote device registered: ${deviceId} at ${remoteIp}`);

    res.json({
      success: true,
      thisDeviceIp,
      message: `Remote device at ${remoteIp} registered successfully on this server`,
    });
  } catch (err) {
    console.error("Device registration error:", err);
    res.status(500).json({
      success: false,
      message: "Device registration failed",
      error: String(err),
    });
  }
});

/**
 * Get list of connected devices
 * GET /api/devices/list
 */
router.get("/list", (_req: Request, res: Response) => {
  try {
    const deviceList = Array.from(connectedDevices.values()).map((d) => ({
      deviceId: d.deviceId,
      deviceIp: d.deviceIp,
      receiveUrl: d.receiveUrl,
      clientType: d.clientType,
      connectedAt: d.connectedAt,
      lastHeartbeat: d.lastHeartbeat,
    }));

    res.json({
      success: true,
      devices: deviceList,
      count: deviceList.length,
    });
  } catch (err) {
    console.error("Get devices error:", err);
    res.status(500).json({
      success: false,
      message: "Failed to get devices",
      error: String(err),
    });
  }
});

/**
 * Send data to another connected device
 * POST /api/devices/:targetDeviceId/send
 * Body: { data: any }
 * Returns: { success: boolean, message: string }
 */
router.post("/:targetDeviceId/send", async (req: Request, res: Response) => {
  try {
    const { targetDeviceId } = req.params;
    const { data } = req.body;

    const targetDevice = connectedDevices.get(targetDeviceId);

    if (!targetDevice) {
      return res.status(404).json({
        success: false,
        message: `Device ${targetDeviceId} not found`,
      });
    }

    if (!data) {
      return res.status(400).json({
        success: false,
        message: "No data provided",
      });
    }

    // Prefer direct delivery to a provided receiveUrl when available
    const directUrls: string[] = [];
    if (targetDevice.receiveUrl) directUrls.push(targetDevice.receiveUrl);
    // fallback to default devices receive path on the device's IP
    directUrls.push(`http://${targetDevice.deviceIp}/api/devices/receive`);

    let delivered = false;
    let lastError: any = null;

    for (const url of directUrls) {
      try {
        const response = await axios.post(url, { data }, { timeout: 5000 });
        targetDevice.lastHeartbeat = new Date();
        delivered = true;
        return res.json({
          success: true,
          message: `Data delivered to ${targetDeviceId} via ${url}`,
          response: response.data,
        });
      } catch (err) {
        lastError = err;
        console.warn(`Delivery to ${targetDeviceId} at ${url} failed:`, err.message || String(err));
        // try next
      }
    }

    // If direct delivery failed (or wasn't possible), queue the message for the target to poll
    (targetDevice.pendingMessages ||= []).push({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      data,
      createdAt: new Date(),
    });

    console.log(`Queued message for ${targetDeviceId} (pending count=${targetDevice.pendingMessages.length})`);

    res.status(202).json({
      success: false,
      queued: true,
      message: `Could not deliver directly to ${targetDeviceId}; message queued for polling`,
      error: lastError ? String(lastError) : undefined,
    });
  } catch (err) {
    console.error("Send data error:", err);
    res.status(500).json({
      success: false,
      message: "Failed to send data",
      error: String(err),
    });
  }
});

/**
 * Receive data from another device
 * POST /api/devices/receive
 * Body: { data: any }
 * Returns: { success: boolean, message: string }
 */
router.post("/receive", (req: Request, res: Response) => {
  try {
    const { data } = req.body;

    if (!data) {
      return res.status(400).json({
        success: false,
        message: "No data received",
      });
    }

    console.log("📥 Data received from device:", data);

    // Process received data here
    // For now, just acknowledge receipt
    res.json({
      success: true,
      message: "Data received successfully",
      receivedAt: new Date(),
    });
  } catch (err) {
    console.error("Receive data error:", err);
    res.status(500).json({
      success: false,
      message: "Failed to receive data",
      error: String(err),
    });
  }
});

/**
 * Get this device's IP address
 * GET /api/devices/info
 */
router.get("/info", (_req: Request, res: Response) => {
  try {
    const thisDeviceIp = getLocalIp();

    res.json({
      success: true,
      deviceId: process.env.DEVICE_ID || "unknown",
      deviceIp: thisDeviceIp,
      connectedDevices: connectedDevices.size,
    });
  } catch (err) {
    console.error("Get info error:", err);
    res.status(500).json({
      success: false,
      message: "Failed to get device info",
      error: String(err),
    });
  }
});

/**
 * Disconnect a device
 * POST /api/devices/:deviceId/disconnect
 */
router.post("/:deviceId/disconnect", (req: Request, res: Response) => {
  try {
    const { deviceId } = req.params;

    const removed = connectedDevices.delete(deviceId);

    if (!removed) {
      return res.status(404).json({
        success: false,
        message: `Device ${deviceId} not found`,
      });
    }

    console.log(`✓ Device disconnected: ${deviceId}`);

    res.json({
      success: true,
      message: `Device ${deviceId} disconnected`,
    });
  } catch (err) {
    console.error("Disconnect error:", err);
    res.status(500).json({
      success: false,
      message: "Failed to disconnect device",
      error: String(err),
    });
  }
});

export default router;

