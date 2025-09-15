import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query) return;
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8001/query', { query });
      setResponse(res.data);
    } catch (err) {
      console.error(err);
      setResponse({error: 'Failed to get response'});
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '800px', margin: '2rem auto', fontFamily: 'Arial, sans-serif' }}>
      <h1>Graph-Powered Conversational Search</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          style={{ width: '70%', padding: '0.5rem', fontSize: '1rem' }}
        />
        <button type="submit" style={{ padding: '0.5rem 1rem', marginLeft: '1rem' }}>Ask</button>
      </form>
      {loading && <p>Loading...</p>}
      {response && (
        <div style={{ marginTop: '2rem' }}>
          {response.error && <p style={{ color: 'red' }}>{response.error}</p>}
          {response.answer && <div><h2>Answer:</h2><p>{response.answer}</p></div>}
          {response.passages && <div><h3>Relevant Passages:</h3><ul>{response.passages.map((p, i)=><li key={i}>{p.text}</li>)}</ul></div>}
          {response.graph && <div><h3>Graph Neighbors:</h3><ul>{response.graph.map((g,i)=><li key={i}><b>{g.entity}</b>: {g.neighbors.join(', ')}</li>)}</ul></div>}
        </div>
      )}
    </div>
  );
}

export default App;