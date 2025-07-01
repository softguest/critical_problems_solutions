import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import Link from 'next/link'
import React from 'react'

const AboutYoollaa = () => {
  return (
    <div className='bg-slate-100 px-4 grid grid-cols-1 min-h-full py-8 lg:px-8 xl:px-32'>
        <Card className='py-8 px-2 md:p-12'>
            <h2 className='text-3xl text-sky-600 font-bold py-2'>Yoollaa: Streamlining Student Admissions and Registration</h2>
            <hr className='my-2 text-sky-600'/>
            <p>Yoollaa is a user-friendly online platform designed to revolutionize the student 
                admissions and registration process for both schools and students. We understand 
                the challenges associated with traditional paper-based registration, and Yoollaa 
                offers a comprehensive solution to streamline the entire process.
            </p>
            <h3 className='text-2xl text-sky-600 font-semibold py-2'>Simplifying Admissions for Schools</h3>
            <p>Yoollaa empowers schools to establish a robust online presence specifically tailored for admissions.  Schools can:</p>
            <ul className='space-y-1 p-4 bg-[#f8f9fb] my-3'>
                <li><strong>Effortlessly create online application forms: </strong>Design and customize online application forms to gather all the necessary student information.</li>
                <li><strong>Manage applications efficiently: </strong> Review and track applications electronically, eliminating the need for physical paperwork.</li>
                <li><strong>Communicate seamlessly with students: </strong>Send automated notifications and updates to students throughout the admissions process.</li>
                
            </ul>
            <h3 className='text-2xl text-sky-600 font-semibold py-2'>Enhancing the Student Experience</h3>
            <p>Yoollaa makes the admissions process smoother and more convenient for students by</p>
            <ul className='space-y-1 p-4 bg-[#f8f9fb] my-3'>
                <li><strong>Providing a user-friendly online application portal: </strong>Students can complete applications from any device at their convenience.</li>
                <li><strong>Eliminating the hassle of paperwork: </strong>Say goodbye to lengthy forms and lost documents.</li>
                <li><strong>Ensuring a faster application turnaround: </strong>Schools can process applications electronically, leading to quicker decisions.</li>
            </ul>
            <h3 className='text-2xl text-sky-600 font-semibold py-2'>Benefits of Yoollaa</h3>
            <p>Yoollaa offers a multitude of advantages for both schools and students:</p>
            <ul className='space-y-1 p-4 bg-[#f8f9fb] my-3'>
                <li><strong>Increased Efficiency: </strong> Save time and resources by eliminating manual processes.</li>
                <li><strong>Improved Organization: </strong>  Maintain a centralized system for all application data.</li>
                <li><strong>Enhanced Communication: </strong>Ensure clear and consistent communication throughout the admissions process.</li>
                <li><strong>Greater Accessibility: </strong> Make admissions information and applications readily available to all students.</li>
            </ul>
            <h3 className='text-2xl text-sky-600 font-semibold py-2'>Join the Future of Admissions with Yoollaa</h3>
            <p>Yoollaa is transforming the way schools manage admissions and student 
                registration.  Let Yoollaa help you create a more efficient 
                and streamlined admissions process for your school, while providing a positive and convenient experience for your students.
            </p>
            <div className='my-6'>
                <Link href='/auth/register'>
                    <Button size="lg" variant={"secondary"} className="">Create An Account</Button>
                </Link>
            </div>
            
        </Card>
    </div>
  )
}

export default AboutYoollaa