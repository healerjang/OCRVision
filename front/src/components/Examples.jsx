import { motion } from 'framer-motion';
import '../styles/QuestionExamples.css';
import '../styles/utils.css'; // ✅ 공통 유틸 스타일 추가

export default function QuestionExamples() {
  return (
    <motion.div
      className="section-wrapper"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      viewport={{ once: true }}
    >
      <h2 className="section-title">예시 이미지</h2>
      <ul className="examples-list">
        <li>명함 속 이름, 이메일, 전화번호 인식</li>
        <li>영수증의 날짜 및 금액 추출</li>
        <li>문서 이미지에서 본문 텍스트 인식</li>
        <li>간판이나 표지판의 문구 추출</li>
      </ul>
    </motion.div>
  );
}
