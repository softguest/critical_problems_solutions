import FingerprintJS from '@fingerprintjs/fingerprintjs';

export const getDeviceId = async (): Promise<string> => {
  const fp = await FingerprintJS.load();
  const result = await fp.get();
  return result.visitorId; // This is a stable unique ID
};