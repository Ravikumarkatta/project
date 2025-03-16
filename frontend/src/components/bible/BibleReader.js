import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchVerseText, setCurrentReference } from '../../state/slices/bibleSlice';
import VerseDisplay from './VerseDisplay';
import BibleControls from './BibleControls';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorAlert from '../common/ErrorAlert';
import VerseContext from './VerseContext';
import CrossReferences from './CrossReferences';
import LexiconPopup from '../lexicon/LexiconPopup';

const BibleReader = ({ initialReference = 'John 3:16' }) => {
  const dispatch = useDispatch();
  const { currentVersion, currentText, loading, error } = useSelector((state) => state.bible);
  const { denominationalPerspective } = useSelector((state) => state.user.preferences);
  const [showContext, setShowContext] = useState(false);
  const [showCrossRefs, setShowCrossRefs] = useState(false);
  const [selectedWord, setSelectedWord] = useState(null);

  useEffect(() => {
    dispatch(setCurrentReference(initialReference));
    dispatch(fetchVerseText({ reference: initialReference, version: currentVersion }));
  }, [dispatch, initialReference, currentVersion]);

  const handleReferenceChange = (newReference) => {
    dispatch(setCurrentReference(newReference));
    dispatch(fetchVerseText({ reference: newReference, version: currentVersion }));
    setShowContext(false);
    setShowCrossRefs(false);
  };

  const handleWordSelect = (word, language) => {
    setSelectedWord({ word, language });
  };

  const closePopups = () => {
    setSelectedWord(null);
  };

  if (loading) return <LoadingSpinner message="Loading Bible text..." />;
  if (error) return <ErrorAlert message={error} />;
  if (!currentText) return <LoadingSpinner message="Loading Bible text..." />;

  return (
    <div className="bible-reader">
      <BibleControls 
        currentReference={initialReference} 
        currentVersion={currentVersion}
        onReferenceChange={handleReferenceChange}
        onToggleContext={() => setShowContext(!showContext)}
        onToggleCrossRefs={() => setShowCrossRefs(!showCrossRefs)}
        showContext={showContext}
        showCrossRefs={showCrossRefs}
      />
      
      <VerseDisplay 
        text={currentText.text} 
        reference={currentText.reference}
        version={currentVersion}
        onWordSelect={handleWordSelect}
      />
      
      {showContext && (
        <VerseContext 
          reference={currentText.reference} 
          version={currentVersion}
          denominationalPerspective={denominationalPerspective}
        />
      )}
      
      {showCrossRefs && (
        <CrossReferences 
          reference={currentText.reference}
          version={currentVersion}
          onReferenceClick={handleReferenceChange}
        />
      )}
      
      {selectedWord && (
        <LexiconPopup
          word={selectedWord.word}
          language={selectedWord.language}
          onClose={closePopups}
        />
      )}
    </div>
  );
};

export default BibleReader;
