import { motion } from 'framer-motion';
import '../styles/FeatureCards.css';
import '../styles/utils.css';

export default function FeatureCards() {
  const features = [
    {
      title: '간편한 사용',
      description: ['이미지 업로드 한 번으로', '텍스트 추출 완료'],
      bg: '/black.png',
    },
    {
      title: 'AI 기반 OCR',
      description: ['최신 딥러닝 모델이', '이미지 속 글자를 인식'],
      bg: '/blue.png',
    },
    {
      title: '다양한 이미지 지원',
      description: ['문서, 명함, 간판 등', '다양한 형식 인식 가능'],
      bg: '/green.png',
    },
    {
      title: '높은 인식 정확도',
      description: ['흐릿한 이미지도', '정확히 텍스트 추출'],
      bg: '/yellow.png',
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
      <section className="feature-section">
        <h2 className="section-title">서비스 특징</h2>
        <div className="feature-grid">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="feature-card"
              style={{ backgroundImage: `url(${feature.bg})` }}
            >
              <h3 className="feature-card-title">{feature.title}</h3>
              <p className="feature-card-desc">
                {Array.isArray(feature.description)
                  ? feature.description.map((line, i) => (
                      <span key={i}>
                        {line}
                        <br />
                      </span>
                    ))
                  : feature.description}
              </p>
            </div>
          ))}
        </div>
      </section>
    </motion.div>
  );
}
