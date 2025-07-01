import { NextApiRequest } from 'next';

export function getClientIp(req: NextApiRequest): string {
  const forwarded = req.headers['x-forwarded-for'];
  return typeof forwarded === 'string' ? forwarded.split(',')[0] : req.socket.remoteAddress || '';
}