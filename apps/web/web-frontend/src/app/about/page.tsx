"use client";
import React from 'react';
import styled from 'styled-components';
import { useRouter } from 'next/navigation';

const AboutWrapper = styled.div`
  min-height: 100vh;
  background-color: #1f1f1f;
  color: #EBE1BD;
  padding: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const TeamGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 30px;
  max-width: 1000px;
  width: 100%;
  margin-top: 40px;
`;

const MemberCard = styled.div`
  background: #262626;
  border: 1px solid #3A4946;
  padding: 20px;
  border-radius: 12px;
  text-align: center;
  transition: transform 0.2s;
  &:hover { transform: translateY(-5px); }
`;

const BackButton = styled.button`
  background: #8fb3a9;
  color: #1f1f1f;
  border: none;
  padding: 12px 30px;
  border-radius: 8px;
  font-weight: bold;
  cursor: pointer;
  margin-bottom: 40px;
  align-self: flex-start;
`;

export default function AboutUs() {
  const router = useRouter();

  const team = [
    { name: "Your Name", role: "Lead Developer", bio: "Fibre analysis specialist." },
    { name: "Group Member 1", role: "Data Scientist", bio: "Visualizations and Statistics." },
    { name: "Group Member 2", role: "UI/UX Designer", bio: "Frontend architecture." },
    // Add more members as needed
  ];

  return (
    <AboutWrapper>
      <BackButton onClick={() => router.push('/')}>← Back to Dashboard</BackButton>
      
      <h1>About Our Group</h1>
      <p style={{ maxWidth: '600px', textAlign: 'center', opacity: 0.8 }}>
        We are a team of researchers and engineers dedicated to improving the 
        quality and consistency of fiber production through data-driven insights.
      </p>

      <TeamGrid>
        {team.map(member => (
          <MemberCard key={member.name}>
            <div style={{ width: '80px', height: '80px', background: '#3A4946', borderRadius: '50%', margin: '0 auto 15px' }} />
            <h3>{member.name}</h3>
            <h5 style={{ color: '#8fb3a9', marginBottom: '10px' }}>{member.role}</h5>
            <p style={{ fontSize: '0.9rem', opacity: 0.7 }}>{member.bio}</p>
          </MemberCard>
        ))}
      </TeamGrid>
    </AboutWrapper>
  );
}