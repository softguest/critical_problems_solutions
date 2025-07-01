import { z } from 'zod';

export const postSchema = z.object({
  title: z.string().min(5, "Title is too short"),
  content: z.string().min(20, "Content must be at least 20 characters long"),
});
