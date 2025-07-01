"use client"
import { auth } from "@/auth";
import Link from "next/link";
import React, { useEffect, useState, useRef } from "react";
import { FaArrowRight, FaCircle } from "react-icons/fa";

type User = {
  id: string;
  firstName?: string;
  lastName?: string;
  email: string;
  role: "ADMIN" | "COACHE" | "PATIENT";
  profileImage: string;
  profession: string;
  Location:   string;
  telephone:  string;
  country:    string;
  address:    string;
  bio:        string;
};

const ExpertList = () => {
    const [theme, setTheme] = useState("light");

  const [users, setUsers] = useState<User[]>([]);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState<string>("ALL");

  useEffect(() => {
    fetch("/api/users")
      .then((res) => res.json())
      .then((data) => setUsers(data));
  }, []);

  const updateRole = async (id: string, role: User["role"]) => {
    await fetch(`/api/users/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    });
    setUsers(users.map((u) => (u.id === id ? { ...u, role } : u)));
  };

  const deleteUser = async (id: string) => {
    await fetch(`/api/users/${id}`, { method: "DELETE" });
    setUsers(users.filter((u) => u.id !== id));
  };

  const filteredUsers = users.filter((user) => {
    const matchesSearch =
      user.email.toLowerCase().includes(search.toLowerCase()) ||
      user.firstName?.toLowerCase().includes(search.toLowerCase());
    const matchesRole = roleFilter === "ALL" || user.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  const isDark = theme === "dark";

  return (
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 xl:grid-cols-4 gap-4 mt-10 px-2 md:px-16">
        {users.map((profile, idx) => (
          <Link key={profile.id} href={`/profile/${profile.id}`}>
            <div
              key={idx}
              className={`rounded-xl p-4 shadow-lg transition hover:scale-[1.02] hover:bg-slate-100 cursor-pointer  ${
                isDark
                  ? "bg-[#2a2b3c] text-white border border-gray-600"
                  : "bg-white text-indigo-900 border border-indigo-100"
              }`}
            >
              <div className="flex justify-between">
                <div>
                  <img
                    src={profile.profileImage}
                    alt={profile.firstName}
                    className="w-16 h-16 rounded-full object-cover mb-2"
                  />
                </div>
                <div>
                  <FaCircle color="#f7f7f7" size={38}/>
                </div>
              </div>
              <h3 className="text-md font-semibold">{profile.firstName} {profile.lastName}</h3>
              <p className="text-sm opacity-80">{profile.role}</p>
              <p className="text-xs opacity-60">
                {/* {profile.email} */}
               Email: *************
              </p>
              <div className="flex justify-between border-t-2 pt-2 mt-1">
                  <div className="text-[12px] flex justify-center items-center space-x-2"><div>Base:</div> <div><FaArrowRight /></div></div>
                  <div className="text-[12px]">{profile?.country}</div>
              </div>
            </div>
          </Link>  
        ))}
      </div>
  )
}

export default ExpertList