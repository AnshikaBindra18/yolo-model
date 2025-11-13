import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000',
});

export async function analyzeVegetation(file) {
  console.log('[Frontend] Sending vegetation analysis request for file:', file.name);
  const form = new FormData();
  form.append('image', file);
  try {
    const { data } = await api.post('/analyze/vegetation', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    console.log('[Frontend] Vegetation analysis response:', data);
    return data; // { vegetation: boolean, mask_image_base64: string|null }
  } catch (err) {
    console.error('[Frontend] Vegetation analysis failed:', err.response?.data || err.message);
    throw err;
  }
}

export async function analyzeSoil(file) {
  const form = new FormData();
  form.append('image', file);
  const { data } = await api.post('/analyze/soil', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data; // { soil_type: string|null, confidence: number }
}
