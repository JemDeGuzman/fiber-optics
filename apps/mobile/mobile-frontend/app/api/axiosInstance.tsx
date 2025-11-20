// src/api/axiosInstance.ts
import axios from "axios";
import Constants from "expo-constants";

const DEV_BASE =
  // recommend setting in expo env or editing below to your machine IP when testing on device
  process.env.EXPO_PUBLIC_API_URL ?? (Constants.manifest?.extra?.API_URL as string) ?? "http://192.168.1.22:4000";

const api = axios.create({
  baseURL: DEV_BASE,
  timeout: 15000,
});

export default api