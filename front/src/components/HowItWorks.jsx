import { motion } from 'framer-motion';
import '../styles/HowItWorks.css';
import '../styles/utils.css';

export default function HowItWorks() {
  const steps = [
    {
      title: '1. 이미지 준비',
      description: '문서, 명함, 간판 등 텍스트가 포함된 이미지를 준비합니다.',
    },
    {
      title: '2. 업로드',
      description: '이미지를 업로드하거나 드래그 앤 드롭으로 올립니다.',
    },
    {
      title: '3. 텍스트 추출',
      description: 'AI가 이미지 속 텍스트를 자동으로 인식합니다.',
    },
    {
      title: '4. 결과 확인',
      description: '추출된 텍스트를 확인하고 복사할 수 있습니다.',
    },
    {
      title: '5. 편집 및 저장',
      description: '필요한 텍스트만 선택하여 저장하거나 다른 곳에 활용합니다.',
    },
    {
      title: '6. 다시 시도',
      description: '다른 이미지를 업로드하여 반복적으로 이용할 수 있습니다.',
    },
  ];

  return (
    <motion.div
      className="section-wrapper"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      viewport={{ once: true }}
    >
      <h2 className="section-title">어떻게 작동하나요?</h2>
      <div className="howitworks-grid">
        {steps.map((step, idx) => (
          <div key={idx} className="howitworks-card">
            <h3 className="howitworks-card-title">{step.title}</h3>
            <p className="howitworks-card-desc">{step.description}</p>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
