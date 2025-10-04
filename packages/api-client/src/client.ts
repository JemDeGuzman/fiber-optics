import axios from "axios";

const API_BASE = process.env.API_URL || "http://localhost:4000/api";
const client = axios.create({ baseURL: API_BASE });

export default client;
