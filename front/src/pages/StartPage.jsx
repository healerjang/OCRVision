// src/pages/StartPage.jsx
import QuestionCard from '../components/UploadPage';

export default function StartPage() {
    return (
        <div className="h-screen overflow-y-scroll snap-y snap-mandatory scroll-smooth bg-[#0e111a] text-white font-sans">
            <section className="h-screen snap-start flex items-center justify-center px-4">
                <QuestionCard
                    question="당신이 생각한 것은 살아있나요?"
                    options={['예', '아니오']}
                />
            </section>
            <section className="h-screen snap-start flex items-center justify-center px-4 bg-[#181b2a]">
                <QuestionCard
                    question="그것은 유명한 사람인가요?"
                    options={['예', '아니오', '잘 모르겠어요']}
                />
            </section>
            <section className="h-screen snap-start flex items-center justify-center px-4 bg-[#222433]">
                <QuestionCard
                    question="그것은 한국인인가요?"
                    options={['예', '아니오']}
                />
            </section>
        </div>
    );
}
