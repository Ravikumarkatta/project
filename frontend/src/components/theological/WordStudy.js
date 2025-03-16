import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { getWordStudy, getWordConcordance } from '../../services/lexiconService';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorAlert from '../common/ErrorAlert';
import BibleReferences from '../bible/BibleReferences';

const WordStudy = ({ word, language }) => {
  const [wordData, setWordData] = useState(null);
  const [concordance, setConcordance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('definition');
  
  const { currentVersion } = useSelector((state) => state.bible);
  
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const wordStudyData = await getWordStudy(word, language);
        const concordanceData = await getWordConcordance(word, language);
        
        setWordData(wordStudyData);
        setConcordance(concordanceData);
      } catch (err) {
        setError(err.message || 'Error fetching word data');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [word, language]);
  
  if (loading) return <LoadingSpinner message="Loading word study..." />;
  if (error) return <ErrorAlert message={error} />;
  if (!wordData) return <ErrorAlert message="No data found for this word" />;
  
  return (
    <div className="word-study">
      <div className="word-header">
        <h2>{wordData.transliteration} ({wordData.original})</h2>
        <div className="pronunciation">{wordData.pronunciation}</div>
      </div>
      
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'definition' ? 'active' : ''}`}
          onClick={() => setActiveTab('definition')}
        >
          Definition
        </button>
        <button
          className={`tab ${activeTab === 'usage' ? 'active' : ''}`}
          onClick={() => setActiveTab('usage')}
        >
          Usage
        </button>
        <button
          className={`tab ${activeTab === 'concordance' ? 'active' : ''}`}
          onClick={() => setActiveTab('concordance')}
        >
          Concordance
        </button>
      </div>
      
      <div className="tab-content">
        {activeTab === 'definition' && (
          <div className="definition-content">
            <h3>Definition</h3>
            <p>{wordData.definition}</p>
            
            <h4>Root</h4>
            <p>{wordData.root || 'N/A'}</p>
            
            <h4>Parts of Speech</h4>
            <p>{wordData.partOfSpeech}</p>
            
            {wordData.semanticDomain && (
              <>
                <h4>Semantic Domain</h4>
                <p>{wordData.semanticDomain}</p>
              </>
            )}
          </div>
        )}
        
        {activeTab === 'usage' && (
          <div className="usage-content">
            <h3>Biblical Usage</h3>
            <p>{wordData.biblicalUsage}</p>
            
            <h4>Key Examples</h4>
            <BibleReferences 
              references={wordData.keyExamples} 
              version={currentVersion}
              showPreview={true}
            />
            
            {wordData.theologicalImplications && (
              <>
                <h4>Theological Implications</h4>
                <p>{wordData.theologicalImplications}</p>
              </>
            )}
          </div>
        )}
        
        {activeTab === 'concordance' && concordance && (
          <div className="concordance-content">
            <h3>Concordance</h3>
            <p>This word appears {concordance.occurrences} times in Scripture.</p>
            
            <div className="occurrence-list">
              {concordance.verses.map((verse, index) => (
                <div key={index} className="occurrence-item">
                  <div className="reference">{verse.reference}</div>
                  <div className="context">{verse.context}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WordStudy;
