import React, { useState, useEffect } from 'react';
import { listBibles, getBible } from './api';

/**
 * App component to display available Bibles and their content.
 */
function App() {
    const [bibles, setBibles] = useState([]);
    const [selectedBible, setSelectedBible] = useState(null);
    const [content, setContent] = useState('');

    // Fetch available Bibles on component mount
    useEffect(() => {
        async function fetchBibles() {
            try {
                const data = await listBibles();
                setBibles(data.bibles);
            } catch (error) {
                console.error('Error fetching bibles:', error);
            }
        }
        fetchBibles();
    }, []);

    // Handle click event to fetch and display selected Bible content
    const handleBibleClick = async (version) => {
        try {
            const data = await getBible(version);
            setSelectedBible(data.version);
            setContent(data.content);
        } catch (error) {
            console.error('Error fetching Bible content:', error);
        }
    };

    return (
        <div>
            <h1>Bible AI</h1>
            <h2>Available Bibles</h2>
            <ul>
                {bibles.map((bible) => (
                    <li key={bible} onClick={() => handleBibleClick(bible)}>
                        {bible}
                    </li>
                ))}
            </ul>
            {selectedBible && (
                <div>
                    <h2>{selectedBible}</h2>
                    <pre>{content}</pre>
                </div>
            )}
        </div>
    );
}

export default App;
