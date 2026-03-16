export const formatNumber = (value: number): string =>
  new Intl.NumberFormat().format(value);

export const formatPercent = (value: number): string => `${value.toFixed(1)}%`;

export const formatTime = (unixSeconds: number): string =>
  new Date(unixSeconds * 1000).toLocaleTimeString();
