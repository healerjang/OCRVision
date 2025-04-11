import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import axios from "axios";

export default function LoadingPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const imageFile = location.state?.imageFile;

  const [fakeProgress, setFakeProgress] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);

  // 퍼센트 애니메이션
  useEffect(() => {
    const interval = setInterval(() => {
      setFakeProgress((prev) => {
        if (prev < 90 && !isCompleted) return prev + 1;
        return prev;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isCompleted]);

  // OCR 요청
  useEffect(() => {
    if (!imageFile) {
      navigate("/upload");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);

    axios
      .post("http://localhost:5000/image/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: "text",
      })
      .then((res) => {
        setIsCompleted(true);

        const lines = res.data.trim().split("\n");

        // ✅ 텍스트가 없는 경우 처리
        const errorLine = lines.find((line) => {
          try {
            const parsed = JSON.parse(line);
            return parsed.statusCode === 1;
          } catch {
            return false;
          }
        });

        if (errorLine) {
          alert("이미지에서 텍스트를 찾을 수 없습니다.");
          navigate("/upload");
          return;
        }

        const lastLine = JSON.parse(lines[lines.length - 1]);
        const ocrBoxes = lastLine.result.map((box) => ({
          text: Array.isArray(box.text) ? box.text.join(" ") : box.text,
          startX: box.start_x,
          startY: box.start_y,
          endX: box.end_x,
          endY: box.end_y,
        }));

        const resultText = ocrBoxes.map((box) => box.text).join("\n");

        setTimeout(() => {
          navigate("/result", {
            state: {
              previewURL: URL.createObjectURL(imageFile),
              ocrBoxes,
              resultText,
            },
          });
        }, 300);
      })
      .catch((err) => {
        console.error("OCR 실패", err);
        alert("OCR 처리 중 오류가 발생했습니다.");
        navigate("/upload");
      });
  }, [imageFile, navigate]);

  return (
    <div className="w-screen h-screen flex flex-col items-center justify-center bg-black/80 text-white">
      <p className="text-xl font-bold mb-4">텍스트를 추출 중입니다...</p>

      {/* 프로그레스 바 */}
      <div className="w-72 h-4 bg-gray-600 rounded-full overflow-hidden">
        <div
          className="bg-indigo-500 h-full transition-all duration-300"
          style={{ width: `${isCompleted ? 100 : fakeProgress}%` }}
        />
      </div>

      {/* 숫자 퍼센트 */}
      <p className="mt-2 text-sm text-gray-300">
        {isCompleted ? "100%" : `${fakeProgress}%`}
      </p>
    </div>
  );
}
