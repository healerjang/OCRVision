import { motion } from 'framer-motion';
import '../styles/FAQSection.css';
import '../styles/utils.css'; // ✅ 공통 유틸 스타일 추가

export default function FAQSection() {
  return (
    <motion.div
      className="section-wrapper"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      viewport={{ once: true }}
    >
      <h2 className="section-title">자주 묻는 질문</h2>
      <div className="faq-wrapper">
        <div>
          <p className="faq-question">Q. 어떤 이미지를 업로드할 수 있나요?</p>
          <p>A. 명함, 문서, 영수증, 간판 등 텍스트가 포함된 이미지를 업로드할 수 있습니다.</p>
        </div>
        <div>
          <p className="faq-question">Q. 어떤 방식으로 텍스트를 추출하나요?</p>
          <p>A. 딥러닝 기반 OCR 엔진이 이미지 내 텍스트 영역을 인식하고 글자를 추출합니다.</p>
        </div>
        <div>
          <p className="faq-question">Q. 모바일에서도 이용할 수 있나요?</p>
          <p>A. 네, 모든 디바이스에서 반응형 UI로 편리하게 사용할 수 있습니다.</p>
        </div>
      </div>
    </motion.div>
  );
}
