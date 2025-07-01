'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

const slides = [
  {
    heading: 'Seek Professional Therapy',
    description:
      'Working with a trained therapist or counselor helps war survivors process traumatic memories, develop healthy coping strategies, and regain a sense of safety and self-worth in a structured, supportive environment.',
    bgColor: 'bg-blue-100',
  },
  {
    heading: 'Join Support Communities',
    description:
      'Support groups create a safe space where trauma survivors can share their experiences, reduce feelings of isolation, and gain strength by connecting with others who understand the emotional toll of war-related experiences.',
    bgColor: 'bg-green-100',
  },
  {
    heading: 'Practice Daily Mindfulness',
    description:
      'Mindfulness practices such as deep breathing, meditation, and body scanning help individuals stay grounded in the present, reduce anxiety and flashbacks, and reestablish a sense of calm in their daily lives after trauma.',
    bgColor: 'bg-yellow-100',
  },
  {
    heading: 'Create a Safe Environment',
    description:
      'Rebuilding a sense of safety is crucial after war. Establishing a secure home, minimizing triggers, and engaging in routines can support emotional regulation and foster healing by reinforcing a feeling of control.',
    bgColor: 'bg-purple-100',
  },
  {
    heading: 'Express Through Art & Writing',
    description:
      'Creative expression through art, journaling, music, or storytelling provides an outlet for emotions that may be difficult to verbalize. These methods promote healing and help survivors reclaim their voice and identity.',
    bgColor: 'bg-pink-100',
  },
];

export default function WarTraumaSlider() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % slides.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className={`mx-auto min-h-[400px] flex justify-center items-center md:mx-16 md:mt-4 p-6 rounded-xl shadow-md transition-all duration-700 ease-in-out ${slides[activeIndex].bgColor}`}
    >
      <div>
        <div className="relative transition-all duration-700 ease-in-out">
          <h2 className="text-3xl font-bold text-indigo-800 mb-2">
            {slides[activeIndex].heading}
          </h2>
          <p className="text-gray-600 mb-4 md:w-[50%]">{slides[activeIndex].description}</p>
          <Link href="/article">
            <Button>More Details</Button>
          </Link>
        </div>
        <div className="flex justify-center mt-4 space-x-2">
          {slides.map((_, idx) => (
            <span
              key={idx}
              className={`w-3 h-3 rounded-full transition-all duration-300 ${
                idx === activeIndex ? 'bg-indigo-600' : 'bg-gray-300'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
