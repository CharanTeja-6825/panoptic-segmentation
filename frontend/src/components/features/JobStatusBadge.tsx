import { Badge } from '@/components/common/Badge';
import type { JobStatusValue } from '@/types/api';

interface JobStatusBadgeProps {
  status: JobStatusValue;
}

export const JobStatusBadge = ({ status }: JobStatusBadgeProps) => {
  if (status === 'done') {
    return <Badge tone="success">done</Badge>;
  }
  if (status === 'failed') {
    return <Badge tone="danger">failed</Badge>;
  }
  if (status === 'processing') {
    return <Badge tone="warning">processing</Badge>;
  }
  return <Badge>uploaded</Badge>;
};
