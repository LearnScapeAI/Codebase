// utils/api.js
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export const fetchRoadmaps = async () => {
  const response = await fetch(`${API_BASE_URL}/roadmaps`);
  return await response.json();
};