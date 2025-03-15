// src/state/store.js - Redux store configuration
import { configureStore } from '@reduxjs/toolkit';
import userReducer from './slices/userSlice';
import bibleReducer from './slices/bibleSlice';
import searchReducer from './slices/searchSlice';
import theologicalReducer from './slices/theologicalSlice';
import lexiconReducer from './slices/lexiconSlice';
import commentaryReducer from './slices/commentarySlice';

export const store = configureStore({
  reducer: {
    user: userReducer,
    bible: bibleReducer,
    search: searchReducer,
    theological: theologicalReducer,
    lexicon: lexiconReducer,
    commentary: commentaryReducer,
  },
});
