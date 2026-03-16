import { create } from 'zustand';

interface CameraControlState {
  cameraIndex: number;
  cameraLabel: string;
  setCameraIndex: (index: number) => void;
  setCameraLabel: (label: string) => void;
}

export const useCameraStore = create<CameraControlState>((set) => ({
  cameraIndex: 0,
  cameraLabel: '',
  setCameraIndex: (cameraIndex) => set({ cameraIndex }),
  setCameraLabel: (cameraLabel) => set({ cameraLabel }),
}));
