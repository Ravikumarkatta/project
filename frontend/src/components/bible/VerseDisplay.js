
// src/components/bible/VerseDisplay.js - Displays Bible verses with highlighting and interactivity
import React from 'react';
import { useSelector } from 'react-redux';
import OriginalWordHighlight from './OriginalWordHighlight';

const VerseDisplay = ({ text, reference, version, onWordSelect }) => {
  const { fontSize } = useSelector((state) => state.user.preferences);
  
  // This would normally require a more complex parsing of the text
  // with original language data from the backend
  const renderText = () => {
    if (!text) return null;
    
    // For demo purposes, this is simplified
    return text.map((verse, index) => (
      <div key={`${reference}-${index}`} className="verse">
        <span className="verse-number">{verse.verseNumber}</span>
        <span 
          className={`verse-text font-size-${fontSize}`}
          dangerouslySetInnerHTML={{ 
            __html: verse.content.replace(
              /\[(\w+):(\w+)\]/g, 
              (_, word, lang) => `<span class="original-word" data-word="${word}" data-lang="${lang}">${word}</span>`
            )
          }} 
          onClick={(e) => {
            const target = e.target;
            if (target.classList.contains('original-word')) {
              onWordSelect(
                target.getAttribute('data-word'),
                target.getAttribute('data-lang')
              );
            }
          }}
        />
      </div>
    ));
  };

  return (
    <div className="verse-display">
      <h2 className="reference-header">{reference} ({version})</h2>
      <div className="verse-content">
        {renderText()}
      </div>
    </div>
  );
};

export default VerseDisplay;
