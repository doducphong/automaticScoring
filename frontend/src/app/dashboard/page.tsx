'use client';

import React, { useEffect, useState } from 'react';
import {
  Box,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  Card,
  CardBody,
  Heading,
  Text,
  Flex,
  Icon,
  useColorModeValue,
  Spinner,
} from '@chakra-ui/react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FiUsers, FiFileText } from 'react-icons/fi';
import api from '@/lib/api';

interface Student {
  id: string;
  name: string;
  student_code: string;
}

interface Exam {
  id: string;
  score: number | null;
  file_format: 'docx' | 'xlsx';
}

export default function Dashboard() {
  const [stats, setStats] = useState({
    totalStudents: 0,
    totalExams: 0,
    docxDistribution: [0, 0, 0],
    xlsxDistribution: [0, 0, 0],
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [studentsRes, examsRes] = await Promise.all([
          api.get<Student[]>('/students/'),
          api.get<Exam[]>('/exams/'),
        ]);

        const students = studentsRes.data;
        const exams = examsRes.data;

        const totalStudents = students.length;
        const totalExams = exams.length;

        const categorize = (score: number | null) => {
          if (score === null) return null;
          if (score < 6.5) return 0;
          if (score < 8) return 1;
          return 2;
        };

        const docxDist = [0, 0, 0];
        const xlsxDist = [0, 0, 0];

        exams.forEach(e => {
          const cat = categorize(e.score);
          if (cat === null) return;
          if (e.file_format === 'docx') docxDist[cat]++;
          if (e.file_format === 'xlsx') xlsxDist[cat]++;
        });

        setStats({
          totalStudents,
          totalExams,
          docxDistribution: docxDist,
          xlsxDistribution: xlsxDist,
        });
      } catch (error) {
        console.error('Error fetching stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const COLORS = ['#E53E3E', '#D69E2E', '#38A169'];
  const LABELS = ['< 6.5', '6.5 - 7.9', '>= 8'];

  const createChartData = (data: number[]) =>
    data.map((value, index) => ({ name: LABELS[index], value }));

  const StatCard = ({
    title,
    value,
    icon,
  }: {
    title: string;
    value: number;
    icon: React.ElementType;
  }) => (
    <Card>
      <CardBody>
        <Flex justifyContent="space-between" alignItems="center">
          <Stat>
            <StatLabel fontWeight="medium" isTruncated>
              {title}
            </StatLabel>
            <StatNumber fontSize="3xl" fontWeight="medium">
              {value}
            </StatNumber>
          </Stat>
          <Box
            p={3}
            bg={useColorModeValue('blue.100', 'blue.900')}
            borderRadius="full"
          >
            <Icon
              as={icon}
              fontSize="2xl"
              color={useColorModeValue('blue.500', 'blue.200')}
            />
          </Box>
        </Flex>
      </CardBody>
    </Card>
  );

  return (
    <Box>
      <Box mb={6}>
        <Heading size="lg" mb={2}>Dashboard</Heading>
        <Text color="gray.600">Welcome to the Teacher Scoring System</Text>
      </Box>

      {loading ? (
        <Flex justify="center" align="center" minH="200px">
          <Spinner size="xl" />
        </Flex>
      ) : (
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={6}>
          <StatCard title="Total Students" value={stats.totalStudents} icon={FiUsers} />
          <StatCard title="Total Exams" value={stats.totalExams} icon={FiFileText} />
        </SimpleGrid>
      )}

      {!loading && (
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
          <Card>
            <CardBody>
              <Heading size="md" mb={4}>DOCX Exam Score Distribution</Heading>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={createChartData(stats.docxDistribution)}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label
                  >
                    {createChartData(stats.docxDistribution).map((_, index) => (
                      <Cell key={`cell-docx-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardBody>
          </Card>

          <Card>
            <CardBody>
              <Heading size="md" mb={4}>XLSX Exam Score Distribution</Heading>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={createChartData(stats.xlsxDistribution)}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label
                  >
                    {createChartData(stats.xlsxDistribution).map((_, index) => (
                      <Cell key={`cell-xlsx-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardBody>
          </Card>
        </SimpleGrid>
      )}

      <Box mt={8}>
        <Heading size="md" mb={4}>Quick Actions</Heading>
        <Card>
          <CardBody>
            <Text>Use the navigation menu on the left to access different functions:</Text>
            <Box as="ul" pl={5} mt={2}>
              <Box as="li" mt={1}><Text><strong>Grading:</strong> Score DOCX and Excel files</Text></Box>
              <Box as="li" mt={1}><Text><strong>Students:</strong> Manage student information</Text></Box>
              <Box as="li" mt={1}><Text><strong>Exams:</strong> Create and manage exam templates</Text></Box>
              <Box as="li" mt={1}><Text><strong>Profile:</strong> Update your account information</Text></Box>
            </Box>
          </CardBody>
        </Card>
      </Box>
    </Box>
  );
}
