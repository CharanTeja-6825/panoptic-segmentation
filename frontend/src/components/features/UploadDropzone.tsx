import type { DragEvent } from 'react';

interface UploadDropzoneProps {
  onSelect: (file: File) => void;
}

const acceptedExtensions = ['.mp4', '.avi', '.mov', '.mkv'];

export const UploadDropzone = ({ onSelect }: UploadDropzoneProps) => {
  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files.item(0);
    if (file) {
      onSelect(file);
    }
  };

  return (
    <label
      className="block cursor-pointer rounded-xl border-2 border-dashed border-slate-700 p-8 text-center transition hover:border-indigo-400"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
      <input
        className="hidden"
        type="file"
        accept={acceptedExtensions.join(',')}
        onChange={(event) => {
          const file = event.target.files?.item(0);
          if (file) {
            onSelect(file);
          }
        }}
      />
      <p className="text-sm font-medium text-slate-100">Drag & drop a video or click to browse</p>
      <p className="mt-1 text-xs text-slate-400">Supported: MP4, AVI, MOV, MKV (max 500MB)</p>
    </label>
  );
};
