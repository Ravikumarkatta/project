
// src/state/slices/bibleSlice.js - Bible state management
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { fetchBibleVersions, fetchBibleText, searchBible } from '../../services/bibleService';

export const initializeBibleVersions = createAsyncThunk(
  'bible/initializeVersions',
  async () => {
    return await fetchBibleVersions();
  }
);

export const fetchVerseText = createAsyncThunk(
  'bible/fetchVerseText',
  async ({ reference, version }) => {
    return await fetchBibleText(reference, version);
  }
);

export const performBibleSearch = createAsyncThunk(
  'bible/search',
  async ({ query, version, filters }) => {
    return await searchBible(query, version, filters);
  }
);

const initialState = {
  availableVersions: [],
  currentVersion: 'NIV',
  currentReference: null,
  currentText: null,
  searchResults: [],
  loading: false,
  error: null,
};

const bibleSlice = createSlice({
  name: 'bible',
  initialState,
  reducers: {
    setCurrentVersion: (state, action) => {
      state.currentVersion = action.payload;
    },
    setCurrentReference: (state, action) => {
      state.currentReference = action.payload;
    },
    clearSearch: (state) => {
      state.searchResults = [];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(initializeBibleVersions.fulfilled, (state, action) => {
        state.availableVersions = action.payload;
      })
      .addCase(fetchVerseText.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchVerseText.fulfilled, (state, action) => {
        state.loading = false;
        state.currentText = action.payload;
      })
      .addCase(fetchVerseText.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      })
      .addCase(performBibleSearch.pending, (state) => {
        state.loading = true;
      })
      .addCase(performBibleSearch.fulfilled, (state, action) => {
        state.loading = false;
        state.searchResults = action.payload;
      })
      .addCase(performBibleSearch.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

export const { setCurrentVersion, setCurrentReference, clearSearch } = bibleSlice.actions;

export default bibleSlice.reducer;

// src/services/bibleService.js - Bible API services
import axios from 'axios';

const API_BASE_URL = '/api';

export const fetchBibleVersions = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/bible/versions`);
    return response.data;
  } catch (error) {
    console.error('Error fetching Bible versions:', error);
    throw error;
  }
};

export const fetchBibleText = async (reference, version = 'NIV') => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/bible/text?reference=${encodeURIComponent(reference)}&version=${version}`
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching Bible text:', error);
    throw error;
  }
};

export const searchBible = async (query, version = 'NIV', filters = {}) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/bible/search`, {
      params: {
        query,
        version,
        ...filters,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error searching Bible:', error);
    throw error;
  }
};

export const getVerseContext = async (reference, version = 'NIV') => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/bible/context?reference=${encodeURIComponent(reference)}&version=${version}`
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching verse context:', error);
    throw error;
  }
};

export const getCrossReferences = async (reference, version = 'NIV') => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/bible/cross-references?reference=${encodeURIComponent(reference)}&version=${version}`
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching cross references:', error);
    throw error;
  }
};
