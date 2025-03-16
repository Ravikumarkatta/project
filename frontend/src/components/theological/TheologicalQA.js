// src/components/theological/TheologicalQA.js - Component for theological Q&A
import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { askQuestion } from '../../state/slices/theologicalSlice';
import { askTheologicalQuestion } from '../../services/theologicalService';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorAlert from '../common/ErrorAlert';
import BibleReferences from '../bible/BibleReferences';

const TheologicalQA = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const { denominationalPerspective } = useSelector((state) => state.user.preferences);
  const { currentVersion } = useSelector((state) => state.bible);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await askTheologicalQuestion(question, denominationalPerspective);
      setResponse(result);
    } catch (err) {
      setError(err.message || 'An error occurred while processing your question.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="theological-qa">
      <h2>Ask a Theological Question</h2>
      
      <form onSubmit={handleSubmit} className="question-form">
        <div className="input-group">
          <label htmlFor="question">Your Question:</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., What does the Bible say about forgiveness?"
            rows={3}
            disabled={loading}
          />
        </div>
        
        <div className="denominational-context">
          <span>Answering from a {denominationalPerspective} perspective</span>
          <a href="/settings" className="change-link">(change)</a>
        </div>
        
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? 'Processing...' : 'Ask Question'}
        </button>
      </form>
      
      {loading && <LoadingSpinner message="Processing your theological question..." />}
      {error && <ErrorAlert message={error} />}
      
      {response && (
        <div className="response-container">
          <h3>Response:</h3>
          <div className="theological-response">
            <p>{response.answer}</p>
            
            {response.scripturalBasis && (
              <div className="scriptural-basis">
                <h4>Scriptural Basis:</h4>
                <BibleReferences 
                  references={response.scripturalBasis} 
                  version={currentVersion}
                />
              </div>
            )}
            
            {response.denominationalNotes && (
              <div className="denominational-notes">
                <h4>Denominational Considerations:</h4>
                <p>{response.denominationalNotes}</p>
              </div>
            )}
            
            {response.furtherReading && (
              <div className="further-reading">
                <h4>Further Reading:</h4>
                <ul>
                  {response.furtherReading.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TheologicalQA;
