import { Navigate, Route, Routes } from 'react-router-dom';

import { AppLayout } from '@/components/layout/AppLayout';
import { ROUTES } from '@/constants/routes';
import { AnalyticsPage } from '@/features/analytics/AnalyticsPage';
import { CameraPage } from '@/features/camera/CameraPage';
import { ChatPage } from '@/features/chat/ChatPage';
import { DashboardPage } from '@/features/dashboard/DashboardPage';
import { ScenePage } from '@/features/scene/ScenePage';
import { VideoPage } from '@/features/video/VideoPage';

export const AppRouter = () => (
  <Routes>
    <Route element={<AppLayout />}>
      <Route path={ROUTES.dashboard} element={<DashboardPage />} />
      <Route path={ROUTES.camera} element={<CameraPage />} />
      <Route path={ROUTES.video} element={<VideoPage />} />
      <Route path={ROUTES.analytics} element={<AnalyticsPage />} />
      <Route path={ROUTES.scene} element={<ScenePage />} />
      <Route path={ROUTES.chat} element={<ChatPage />} />
    </Route>
    <Route path="*" element={<Navigate to={ROUTES.dashboard} replace />} />
  </Routes>
);
