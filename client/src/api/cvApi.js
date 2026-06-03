import axios from 'axios';

const API_BASE =
  import.meta.env.VITE_API_URL ||
  '/api';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

export const analyzeCV = async (
  file,
  onProgress = () => {}
) => {
  try {
    const formData = new FormData();

    formData.append('cv_file', file);

    const response = await apiClient.post(
      '/analyze',
      formData,
      {
        headers: {
          'Content-Type':
            'multipart/form-data',
        },

        onUploadProgress: (e) => {
          if (e.total) {
            onProgress(
              Math.round(
                (e.loaded / e.total) * 100
              )
            );
          }
        },
      }
    );

    return response.data;
  } catch (error) {
    console.error(
      'Analyze CV Error:',
      error
    );

    // Backend merespons error
    if (error.response) {
      throw new Error(
        error.response.data?.error ||
          'Server gagal memproses CV'
      );
    }

    // Request terkirim tapi tidak ada response
    if (error.request) {
      throw new Error(
        'Server tidak merespons'
      );
    }

    // Error lainnya
    throw new Error(
      'Terjadi kesalahan saat upload CV'
    );
  }
};