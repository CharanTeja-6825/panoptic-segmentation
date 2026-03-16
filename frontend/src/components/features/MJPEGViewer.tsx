import { useState } from 'react';

import { Card } from '@/components/common/Card';

interface MJPEGViewerProps {
  src: string;
}

export const MJPEGViewer = ({ src }: MJPEGViewerProps) => {
  const [refreshSeed, setRefreshSeed] = useState(0);

  return (
    <Card className="overflow-hidden p-0">
      <img
        src={`${src}?refresh=${refreshSeed}`}
        alt="Live camera stream"
        className="aspect-video w-full bg-black object-contain"
        onError={() => setRefreshSeed((value) => value + 1)}
      />
    </Card>
  );
};
