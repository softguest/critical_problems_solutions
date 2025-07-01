export type EditorBlock = {
  id: string;
  type: string;
  data: any;
};

// type EditorContent = {
//   time: number;
//   blocks: {
//     id: string;
//     type: string;
//     data: any;
//   }[];
//   version: string;
// };


export type EditorContent = {
  time: number;
  blocks: EditorBlock[];
  version: string;
};