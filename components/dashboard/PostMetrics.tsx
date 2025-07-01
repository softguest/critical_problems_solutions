"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type Metrics = {
  impressions?: number;
  likes?: number;
  comments?: number;
  shares?: number;
  engagement?: number;
};

interface PostMetricsProps {
  metrics: Record<string, any>;
}

export function PostMetrics({ metrics }: PostMetricsProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      <MetricCard label="Impressions" value={metrics.impressions} />
      <MetricCard label="Likes" value={metrics.likes} />
      <MetricCard label="Comments" value={metrics.comments} />
      <MetricCard label="Shares" value={metrics.shares} />
      <MetricCard label="Engagement" value={metrics.engagement} />
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value?: number }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent className="text-2xl font-semibold">{value ?? 0}</CardContent>
    </Card>
  );
}
