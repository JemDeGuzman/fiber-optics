import AsyncStorage from "@react-native-async-storage/async-storage";
import { API_URL } from "../config/API_keys";

export const api = (path: string) => `${API_URL}${path}`;

export async function authFetch(path: string, options: any = {}) {
  const token = await AsyncStorage.getItem("token");
  return fetch(api(path), {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
      Authorization: token ? `Bearer ${token}` : "",
    },
  });
}

export async function loginUser(email: string, password: string) {
  const res = await fetch(api("/api/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return res.json();
}

export async function registerUser(
  name: string,
  email: string,
  password: string
) {
  const res = await fetch(api("/api/auth/register"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
  return res.json();
}

export async function getUser(token: string) {
  const res = await fetch(api("/api/user/me"), {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
}

export async function getSettings(token: string) {
  const res = await fetch(api("/api/user/settings"), {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
}

export async function updateSettings(token: string, settings: any) {
  const res = await fetch(api("/api/user/settings"), {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(settings),
  });
  return res.json();
}
