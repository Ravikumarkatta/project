fetch('/config/frontend_config.json')
  .then(response => response.json())
  .then(config => {
    // Initialize app with config
  });
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

ReactDOM.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
    document.getElementById('root')
);