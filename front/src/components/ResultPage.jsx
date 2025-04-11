import { useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import "../styles/utils.css";
import "../styles/ResultPage.css";

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();

  const resultText = location.state?.resultText || "";
  const previewURL = location.state?.previewURL || "";
  const ocrBoxes = location.state?.ocrBoxes || [];

  const [selectedTextIndex, setSelectedTextIndex] = useState(0);

  const imageRef = useRef(null);
  const containerRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 });
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const updateSize = () => {
    if (imageRef.current && containerRef.current) {
      const imageRect = imageRef.current.getBoundingClientRect();
      const containerRect = containerRef.current.getBoundingClientRect();

      setImageSize({
        width: imageRect.width,
        height: imageRect.height,
      });

      setOffset({
        x: imageRect.left - containerRect.left,
        y: imageRect.top - containerRect.top,
      });
    }
  };

  useEffect(() => {
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [previewURL]);

  const handleRetry = () => navigate("/upload");

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(resultText);
      alert("텍스트가 복사되었습니다!");
    } catch {
      alert("복사에 실패했습니다.");
    }
  };

  const handleDownload = () => {
    const blob = new Blob([resultText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ocr_result.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative w-screen h-screen overflow-auto">
      {/* 배경 */}
      <img src="/bg3.png" alt="bg" className="absolute inset-0 w-full h-full object-cover z-0" />
      <div className="absolute inset-0 bg-black/40 backdrop-blur z-10" />

      {/* 콘텐츠 */}
      <motion.div
        className="relative z-20 flex flex-col items-center justify-center min-h-screen text-white px-4 text-center py-10"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl md:text-3xl font-bold mb-6 drop-shadow-lg">
          텍스트 추출 결과
        </h2>

        <div className="result-wrapper flex flex-col md:flex-row gap-6 items-start max-w-6xl w-full">
          {/* 이미지 + OCR 박스 */}
          <div
            className="result-image-box relative mx-auto"
            ref={containerRef}
          >
            {previewURL ? (
              <>
                <img
                  ref={imageRef}
                  src={previewURL}
                  alt="업로드한 이미지"
                  className="rounded-xl shadow-md max-w-full max-h-[65vh] object-contain mx-auto"
                  onLoad={updateSize}
                />

                {ocrBoxes.map((box, idx) => {
                  const naturalWidth = imageRef.current?.naturalWidth || 1;
                  const naturalHeight = imageRef.current?.naturalHeight || 1;
                  const xRatio = imageSize.width / naturalWidth;
                  const yRatio = imageSize.height / naturalHeight;

                  const left = box.startX * xRatio + offset.x;
                  const top = box.startY * yRatio + offset.y;
                  const width = (box.endX - box.startX) * xRatio;
                  const height = (box.endY - box.startY) * yRatio;

                  return (
                    <div
                      key={idx}
                      className="absolute border border-yellow-400 bg-yellow-300/40 text-black text-xs font-semibold px-1 py-0.5 rounded"
                      style={{
                        top: `${top}px`,
                        left: `${left}px`,
                        width: `${width}px`,
                        height: `${height}px`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        textAlign: "center",
                        userSelect: "none",
                      }}
                    >
                      <span style={{ pointerEvents: "auto", userSelect: "text" }}>
                        {box.text}
                      </span>
                    </div>
                  );
                })}
              </>
            ) : (
              <p className="text-gray-300">이미지가 없습니다.</p>
            )}
          </div>

          {/* 드롭다운 + 선택된 텍스트 */}
          <div className="result-text-box w-full md:w-1/3 text-left">
            {ocrBoxes.length > 0 ? (
              <>
                <label htmlFor="text-select" className="block mb-2 text-sm text-gray-300">
                  텍스트 선택:
                </label>
                <select
                  id="text-select"
                  className="w-full p-2 rounded bg-gray-800 text-white mb-4"
                  value={selectedTextIndex}
                  onChange={(e) => setSelectedTextIndex(Number(e.target.value))}
                >
                  {ocrBoxes.map((box, idx) => (
                    <option key={idx} value={idx}>
                      {box.text.length > 20 ? box.text.slice(0, 20) + "..." : box.text}
                    </option>
                  ))}
                </select>

                <div className="text-sm text-gray-100 whitespace-pre-wrap">
                  {ocrBoxes[selectedTextIndex]?.text}
                </div>
              </>
            ) : (
              <p className="text-gray-300">추출된 텍스트가 없습니다.</p>
            )}
          </div>
        </div>

        {/* 버튼 */}
        <div className="flex flex-col md:flex-row gap-4 mt-6">
          <button className="primary-button btn-indigo" onClick={handleCopy}>
            텍스트 복사하기
          </button>
          <button className="primary-button btn-indigo" onClick={handleDownload}>
            TXT로 다운로드
          </button>
          <button className="primary-button btn-gray" onClick={handleRetry}>
            다른 이미지 업로드하기
          </button>
        </div>
      </motion.div>
    </div>
  );
}
