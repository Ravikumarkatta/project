import axios from 'axios';

const API_BASE_URL = '/api';

export const getHebrewLexicon = async (word) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/lexicon/hebrew`, {
      params: {
        word,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching Hebrew lexicon:', error);
    throw error;
  }
};

export const getGreekLexicon = async (word) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/lexicon/greek`, {
      params: {
        word,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching Greek lexicon:', error);
    throw error;
  }
};

export const getWordStudy = async (word, language) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/lexicon/word-study`, {
      params: {
        word,
        language,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching word study:', error);
    throw error;
  }
};

export const getWordConcordance = async (word, language) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/lexicon/concordance`, {
      params: {
        word,
        language,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching word concordance:', error);
    throw error;
  }
};
