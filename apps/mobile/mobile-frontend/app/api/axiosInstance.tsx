// src/api/axiosInstance.ts
import axios from "axios";
import Constants from "expo-constants";

const DEV_BASE =
  // machine IP when testing 
  process.env.EXPO_PUBLIC_API_URL ?? (Constants.manifest?.extra?.API_URL as string) ?? "http://192.168.1.22:4000";

const api = axios.create({
  baseURL: DEV_BASE,
  timeout: 15000,
});

export default api