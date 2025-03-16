import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars, faSearch, faCog, faBook, faQuestionCircle, faLanguage, faCommentAlt } from '@fortawesome/free-solid-svg-icons';
import MainSearch from './MainSearch';

const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const { theme } = useSelector((state) => state.user.preferences);
  const navigate = useNavigate();

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
    if (searchOpen) setSearchOpen(false);
  };

  const toggleSearch = () => {
    setSearchOpen(!searchOpen);
    if (menuOpen) setMenuOpen(false);
  };

  const handleSearchSubmit = (query) => {
    navigate(`/study?search=${encodeURIComponent(query)}`);
    setSearchOpen(false);
  };

  return (
    <header className={`app-header ${theme}`}>
      <div className="header-container">
        <div className="header-left">
          <button className="menu-toggle" onClick={toggleMenu} aria-label="Toggle menu">
            <FontAwesomeIcon icon={faBars} />
          </button>
          <Link to="/" className="logo">
            Bible-AI
          </Link>
        </div>

        <div className="header-right">
          <button className="search-toggle" onClick={toggleSearch} aria-label="Toggle search">
            <FontAwesomeIcon icon={faSearch} />
          </button>
          <Link to="/settings" className="settings-link">
            <FontAwesomeIcon icon={faCog} />
          </Link>
