import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import '../styles/utils.css';
import '../styles/UploadPage.css';

export default function UploadPage() {
  const [imageFile, setImageFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const navigate = useNavigate();

  const handleImageChange = (file) => {
    if (file) {
      setImageFile(file);
      setPreviewURL(URL.createObjectURL(file));
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    handleImageChange(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    handleImageChange(file);
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleDragEnter = () => setIsDragOver(true);
  const handleDragLeave = () => setIsDragOver(false);

  const handleExtract = () => {
    if (!imageFile) return alert("이미지를 먼저 업로드해주세요!");

    // 로딩 페이지로 전환하며 이미지 전달
    navigate("/loading", {
      state: { imageFile },
    });
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden">
      <img
        src="/bg3.png"
        alt="배경"
        className="absolute inset-0 w-full h-full object-cover pointer-events-none z-0"
      />
      <div className="absolute inset-0 bg-black/40 backdrop-blur z-10" />

      <motion.div
        className="relative z-20 flex flex-col items-center justify-center h-full text-white px-4 text-center"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-3xl font-bold mb-4">이미지 업로드</h1>
        <p className="text-gray-300 mb-6">텍스트를 추출할 이미지를 업로드하세요.</p>

        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          className={`mb-4 w-72 h-32 border-2 rounded-lg border-dashed flex items-center justify-center cursor-pointer transition ${
            isDragOver ? 'border-indigo-400 bg-white/10' : 'border-gray-300'
          }`}
          onClick={() => document.getElementById('file-upload').click()}
        >
          <p className="text-gray-300 text-sm">
            여기로 드래그하거나 클릭해서 이미지 선택
          </p>
        </div>

        <label htmlFor="file-upload" className="file-upload-button mb-3">
          파일 선택
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
        />
        <p className="file-name-label mb-6">
          {imageFile ? imageFile.name : '선택된 파일 없음'}
        </p>

        {previewURL && (
          <div className="mb-6">
            <img
              src={previewURL}
              alt="미리보기"
              className="w-64 h-auto rounded-lg shadow-lg"
            />
          </div>
        )}

        <button
          className="primary-button btn-indigo"
          onClick={handleExtract}
          disabled={!imageFile}
        >
          텍스트 추출하기
        </button>
      </motion.div>
    </div>
  );
}
