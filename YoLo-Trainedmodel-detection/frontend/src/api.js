import axios from 'axios';

const baseUrl = 'http://localhost:8000';

export const detectImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await axios.post(`${baseUrl}/detect/`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    console.error('Detection failed:', error);
    return null;
  }
};
