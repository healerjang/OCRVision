import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import '../styles/CTASection.css';
import '../styles/utils.css'; // ✅ 공통 유틸 스타일 추가

export default function CTASection() {
  const navigate = useNavigate();

  return (
    <motion.div
      className="section-wrapper"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      viewport={{ once: true }}
    >
      <h2 className="text-3xl md:text-4xl font-bold mb-6">
        이미지 속 글자를 추출해보세요
      </h2>
      <p className="text-lg text-gray-300 mb-8">
        사진 하나로 손쉽게 텍스트를 추출해보세요.<br />
        지금 바로 AI OCR 기능을 체험해보세요!
      </p>
      <button
        onClick={() => navigate('/upload')}
        className="primary-button btn-indigo"
      >
        시작하기
      </button>
    </motion.div>
  );
}
