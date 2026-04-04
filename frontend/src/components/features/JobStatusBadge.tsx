import { Badge } from '@/components/common/Badge';
import type { JobStatusValue } from '@/types/api';

interface JobStatusBadgeProps {
  status: JobStatusValue;
}

export const JobStatusBadge = ({ status }: JobStatusBadgeProps) => {
  if (status === 'done') {
    return <Badge variant="success">done</Badge>;
  }
  if (status === 'failed') {
    return <Badge variant="destructive">failed</Badge>;
  }
  if (status === 'processing') {
    return <Badge variant="warning">processing</Badge>;
  }
  return <Badge variant="secondary">uploaded</Badge>;
};
