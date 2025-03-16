
// src/components/commentary/CommentaryView.js - Displays Bible commentaries
import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { getCommentary } from '../../services/commentaryService';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorAlert from '../common/ErrorAlert';

const CommentaryView = ({ reference }) => {
  const [commentaries, setCommentaries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCommentary, setSelectedCommentary] = useState(null);
  
  const { denominationalPerspective } = useSelector((state) => state.user.preferences);
  
  useEffect(() => {
    const fetchCommentaries = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const result = await getCommentary(reference, denominationalPerspective);
        setCommentaries(result);
        
        if (result.length > 0) {
          setSelectedCommentary(result[0]);
        }
      } catch (err) {
        setError(err.message || 'Error fetching commentaries');
      } finally {
        setLoading(false);
      }
    };
    
    fetchCommentaries();
  }, [reference, denominationalPerspective]);
  
  if (loading) return <LoadingSpinner message="Loading commentaries..." />;
  if (error) return <ErrorAlert message={error} />;
  if (commentaries.length === 0) return <p>No commentaries available for this passage.</p>;
  
  return (
    <div
