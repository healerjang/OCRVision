import MainPage from '../components/MainPage';
import Examples from '../components/Examples';
import CTASection from '../components/CTASection';
import FAQSection from '../components/FAQSection';
import HowItWorks from '../components/HowItWorks';
import FeatureCards from '../components/FeatureCards';

export default function MainScrollPage() {
  return (
    <section className="relative w-screen min-h-[500vh] overflow-hidden bg-black">
      {/* ✅ 한 장짜리 배경 이미지 */}
      <img
        src="/bg.jpg" // 하나로 이어진 전체 배경 이미지
        alt="Full Background"
        className="absolute inset-0 w-full h-full object-cover opacity-30 blur-sm z-0 pointer-events-none"
      />

      {/* ✅ 콘텐츠 오버레이 */}
      <div className="relative z-10 text-white font-sans scroll-smooth">
        <section id="home" className="min-h-screen flex items-center justify-center">
          <MainPage />
        </section>

        <section id="intro" className="min-h-screen flex flex-col justify-center space-y-32 py-24">
          <FeatureCards />
          <HowItWorks />
        </section>

        <section id="examples" className="min-h-screen flex items-center justify-center py-24">
          <Examples />
        </section>

        <section id="faq" className="min-h-screen flex items-center justify-center py-24">
          <FAQSection />
        </section>

        <section id="cta" className="min-h-screen flex items-center justify-center py-24">
          <CTASection />
        </section>
      </div>
    </section>
  );
}
