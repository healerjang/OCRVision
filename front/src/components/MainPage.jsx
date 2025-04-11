import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import '../styles/MainPage.css';
import '../styles/utils.css';

export default function MainPage() {
  const navigate = useNavigate();
  const [isExiting, setIsExiting] = useState(false);

  const handleStart = () => {
    setIsExiting(true);
    setTimeout(() => {
      navigate('/upload'); // 나중에 /upload 등으로 변경 가능
    }, 400);
  };

  return (
    <AnimatePresence>
      {!isExiting && (
        <motion.section
          className="section-wrapper"
          initial={{ opacity: 1 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
        >
          <h1 className="main-title text-white">
            AI가 이미지 속 텍스트를 읽어드립니다
          </h1>
          <p className="main-description text-gray-300">
            사진 속 문서, 명함, 간판, 손글씨까지!<br />
            이미지를 업로드하면 인공지능이 텍스트를 추출해 보여드립니다.
          </p>

          <button
            onClick={handleStart}
            className="primary-button btn-indigo mt-6"
          >
            시작하기
          </button>
        </motion.section>
      )}
    </AnimatePresence>
  );
}
