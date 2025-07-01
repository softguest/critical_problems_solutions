'use client';
import { Session } from "next-auth"
import axios from 'axios';
import { useRouter } from 'next/navigation';
import React, { ChangeEvent, FC, useState } from 'react';

interface FormSolutionProps {
  problemId: string;
  data:      Session | null;
}
const FormSolution: FC<FormSolutionProps> = ({ problemId, data }) => {
  const [title, setTitle] = useState<string>('');
  const [content, setContent] = useState<string>('');
  const router = useRouter();
  // const { data } = useSession();

  const handleTitleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setTitle(e.target.value);
  };

  const handleContentChange = (e: ChangeEvent<HTMLInputElement>) => {
    setContent(e.target.value);
  };

  const handleSubmitSolution = async () => {
    if (title.trim() !== '') {
      try {
        const newSolution = await axios.post('/api/solution', {
          problemId,
          title: title,
          content: content
        });
        if (newSolution.status === 200) {
          router.refresh();
        }
      } catch (error) {
        console.error(error);
      }
    }
  };

  return (
    <div>
      <div className='mt-4'>
        <label
          htmlFor='comment'
          className='block text-gray-700 text-sm font-bold mb-2'
        >
          Add Solution Title
        </label>
        <input
          value={title}
          onChange={handleTitleChange}
          type='text'
          className='w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring focus:border-blue-300'
          name='title'
        />
        <label
          htmlFor='comment'
          className='block text-gray-700 text-sm font-bold mb-2'
        >
          Add Solution Content
        </label>
        <input
          value={content}
          onChange={handleContentChange}
          type='text'
          className='w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring focus:border-blue-300'
          name='content'
        />
        <button
          disabled={!data?.user?.email}
          onClick={handleSubmitSolution}
          className='bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-md mt-2 disabled:bg-gray-400'
        >
          Submit Solution
        </button>
      </div>
    </div>
  );
};

export default FormSolution;
