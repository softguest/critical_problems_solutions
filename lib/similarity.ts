// export function cosineSimilarity(vecA: number[], vecB: number[]): number {
//   const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
//   const magA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
//   const magB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
//   return dotProduct / (magA * magB);
// }

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) {
    console.warn("Invalid vectors for similarity comparison", { vecA, vecB });
    return 0;
  }

  const dotProduct = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((acc, val) => acc + val * val, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((acc, val) => acc + val * val, 0));
  return magnitudeA && magnitudeB ? dotProduct / (magnitudeA * magnitudeB) : 0;
}

