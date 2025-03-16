import axios from 'axios';

const API_BASE_URL = '/api';

export const askTheologicalQuestion = async (question, denominationalContext = null) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/theological/ask`, {
      question,
      denominationalContext,
    });
    return response.data;
  } catch (error) {
    console.error('Error asking theological question:', error);
    throw error;
  }
};

export const getDoctrinalSummary = async (topic, denominationalContext = null) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/theological/doctrine`, {
      params: {
        topic,
        denominationalContext,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching doctrinal summary:', error);
    throw error;
  }
};

export const getTheologicalPerspectives = async (topic) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/theological/perspectives`, {
      params: {
        topic,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching theological perspectives:', error);
    throw error;
  }
};
