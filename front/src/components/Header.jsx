// components/Header.jsx
import { useNavigate, useLocation } from 'react-router-dom';
import '../styles/Header.css';
import '../styles/utils.css'; 

export default function Header() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleGoHome = () => {
    if (location.pathname === '/') {
      const target = document.getElementById('home');
      if (target) {
        target.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      navigate('/');
    }
  };

  return (
    <header className="header-wrapper">
      <div className="header-title">
        <button onClick={handleGoHome}>Textify</button>
      </div>

      <nav className="header-nav">
        <button onClick={handleGoHome} className="header-link">홈</button>
        <button
          onClick={() => {
            const el = document.getElementById('intro');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
          }}
          className="header-link"
        >
          앱 소개
        </button>
        <button
          onClick={() => {
            const el = document.getElementById('examples');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
          }}
          className="header-link"
        >
          예시
        </button>
        <button
          onClick={() => {
            const el = document.getElementById('faq');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
          }}
          className="header-link"
        >
          FAQ
        </button>
      </nav>
    </header>
  );
}
