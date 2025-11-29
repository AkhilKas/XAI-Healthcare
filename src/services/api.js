import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

// API methods
export const patientAPI = {
  getList: () => api.get('/patients'),
  getData: (id) => api.get(`/patients/${id}`),
  getMotion: (id, task) => api.get(`/patients/${id}/motion/${task}`),
};

export const analysisAPI = {
  analyze: (patientId, taskId) => api.post('/analysis', { patientId, taskId }),
  getExplanations: (id) => api.get(`/analysis/${id}/explanations`),
  getCounterfactuals: (id) => api.get(`/analysis/${id}/counterfactuals`),
  getRisk: (id) => api.get(`/analysis/${id}/risk`),
};

export const chatAPI = {
  sendMessage: (message, context) => api.post('/chat', { message, context }),
};

export default api;
