"use client";
import React from 'react';
import styled, { createGlobalStyle } from "styled-components";
import { useRouter } from 'next/navigation';

// Main Container
const PageWrapper = styled.div`
  min-height: 100vh;
  background-color: #1f1f1f;
  color: #EBE1BD;
  padding: 40px 20px;
  font-family: sans-serif;
  margin: 0;
`;

const GlobalReset = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: sans-serif;
  }
`;

// Navigation Header
const NavHeader = styled.div`
  max-width: 1200px;
  margin: 0 auto 0 auto;
`;

const ReturnButton = styled.button`
  background-color: #3A4946;
  color: #EBE1BD;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;

  &:hover {
    background-color: #EBE1BD;
    color: #3A4946;
  }
`;

// Content Section
const Content = styled.div`
  max-width: 1000px;
  margin: 0 auto;
  text-align: center;
`;

const GroupGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 24px;
  margin-top: 50px;
`;

const MemberCard = styled.div`
  background: #262626;
  border: 1px solid #3A4946;
  border-radius: 15px;
  padding: 30px;
  transition: border-color 0.3s;

  &:hover {
    border-color: #8fb3a9;
  }
`;

const AvatarCircle = styled.div`
  width: 100px;
  height: 100px;
  background: #3A4946;
  border-radius: 50%;
  margin: 0 auto 20px auto;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  color: #8fb3a9;
`;

const ProfileImage = styled.img`
  width: 120px;
  height: 120px;
  border-radius: 50%;
  margin: 0 auto 20px auto;
  display: block;
  object-fit: cover; /* This prevents the image from stretching */
  border: 3px solid #3A4946;
  transition: border-color 0.3s ease;

  &:hover {
    border-color: #8fb3a9;
  }
`;

export default function AboutPage() {
  const router = useRouter();

  // Temporary Data Structure for you to edit later
  const groupMembers = [
    { id: 1, name: "Jemuel Endrew De Guzman", role: "Team Leader//Data Science", description: "Responsible for team planning, deep learning model development, and project methodology.", imageUrl: "/assets/jedg.png" },
    { id: 2, name: "Mark Laurence Castillo", role: "Prototype Design//Cyber-Physical Systems", description: "Responsible for physical Prototype development, Component integration, and Quality Assurance.", imageUrl: "/assets/mlc.png" },
    { id: 3, name: "Rob Andre Catapang", role: "Software Design//System Administration", description: "Responsible for developing and managing the frontend and backend of the web application", imageUrl: "/assets/rac.png"},
    { id: 4, name: "John Chester Irylle Tayam", role: "Documentation//Cyber-Physical Systems", description: "Responsible for maintaining the project documentation and managing project deliverables.", imageUrl: "/assets/jict.png" },
    { id: 5, name: "Steven Dale Pajarillo", role: "Resource Manager//Data Science", description: "Responsible for contacting the client and gathering resources for the project.", imageUrl: "/assets/sdp.png" },
    { id: 6, name: "Engr. Angielyn Jebulan", role: "Project Design Adviser", description: "Responsible for keeping team on schedule, as well as providing design guidance and feedback.", imageUrl: "/assets/ejl.png" },
  ];

  return (
    <PageWrapper>
      <GlobalReset />
      <NavHeader>
        <ReturnButton onClick={() => router.push('/dashboard')}>
          Return to Dashboard
        </ReturnButton>
      </NavHeader>

      <Content>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '16px' }}>Meet the Team</h1>
        <p style={{ opacity: 0.8, maxWidth: '900px', margin: '0 auto' }}>
          The Fiber Optics Abaca Classification Tracking System was developed by 
          Team 22 of the CpE Design Project 2 Course, SY 2025-2026 at the
          Technological Institute of the Philippines - Quezon City.
        </p>

        <GroupGrid>
          {groupMembers.map((member) => (
            <MemberCard key={member.id}>
              {member.imageUrl ? (
                <ProfileImage 
                  src={member.imageUrl} 
                  alt={member.name} 
                  onError={(e) => {
                    // Fallback to a generic placeholder if image is missing
                    (e.target as HTMLImageElement).src = "https://via.placeholder.com/120";
                  }}
                />
              ) : (
                /* Your old AvatarCircle as a secondary fallback */
                <AvatarCircle>{member.name.charAt(0)}</AvatarCircle>
              )}

              <h3 style={{ margin: '10px 0' }}>{member.name}</h3>
              {member.role.split('//').map((line, index) => (
                <div key={index} style={{ color: '#8fb3a9', fontWeight: 'bold' }}>
                  {line.trim()}
                </div>
              ))}
              <p style={{ fontSize: '0.9rem', lineHeight: '1.5', opacity: 0.7 }}>
                {member.description}
              </p>
            </MemberCard>
          ))}
        </GroupGrid>
      </Content>
    </PageWrapper>
  );
}