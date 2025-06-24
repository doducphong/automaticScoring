'use client';

import React, { ReactNode } from 'react';
import {
  Box,
  Flex,
  Icon,
  Text,
  VStack,
  HStack,
  Heading,
  Avatar,
  Divider,
  useDisclosure,
  Drawer,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  DrawerHeader,
  DrawerBody,
  IconButton,
  useBreakpointValue,
} from '@chakra-ui/react';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import NextLink from 'next/link';
import { 
  FiHome, 
  FiClipboard, 
  FiUsers, 
  FiFileText, 
  FiUser, 
  FiLogOut,
  FiMenu,
} from 'react-icons/fi';

interface SidebarItemProps {
  icon: React.ElementType;
  label: string;
  path: string;
  active: boolean;
}

const SidebarItem = ({ icon, label, path, active }: SidebarItemProps) => {
  return (
    <NextLink href={path} passHref>
      <Flex
        align="center"
        p="4"
        mx="4"
        borderRadius="lg"
        role="group"
        cursor="pointer"
        bg={active ? 'blue.500' : 'transparent'}
        color={active ? 'white' : 'gray.600'}
        _hover={{
          bg: active ? 'blue.600' : 'blue.50',
          color: active ? 'white' : 'blue.600',
        }}
      >
        <Icon
          mr="4"
          fontSize="16"
          as={icon}
        />
        <Text fontWeight={active ? 'medium' : 'normal'}>{label}</Text>
      </Flex>
    </NextLink>
  );
};

export default function DashboardLayout({
  children,
}: {
  children: ReactNode;
}) {
  const { user, logout } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  const isDesktop = useBreakpointValue({ base: false, md: true });

  const sidebarItems = [
    { icon: FiHome, label: 'Dashboard', path: '/dashboard' },
    { icon: FiClipboard, label: 'Grading', path: '/dashboard/grading' },
    { icon: FiUsers, label: 'Students', path: '/dashboard/students' },
    { icon: FiFileText, label: 'Exams', path: '/dashboard/exams' },
    { icon: FiUser, label: 'Profile', path: '/dashboard/profile' },
  ];

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  const SidebarContent = () => (
    <Box
      borderRight="1px"
      borderRightColor="gray.200"
      w={{ base: 'full', md: 60 }}
      pos="fixed"
      h="full"
      bg="white"
    >
      <Flex h="20" alignItems="center" mx="8" justifyContent="space-between">
        <Heading fontSize="2xl" fontWeight="bold" color="blue.600">
          Teacher Portal
        </Heading>
      </Flex>
      
      {user && (
        <Flex px="8" mb="6" direction="column" alignItems="center">
          <Avatar size="md" name={user.full_name} mb="2" />
          <Text fontWeight="medium" textAlign="center">{user.full_name}</Text>
          <Text fontSize="sm" color="gray.500" textAlign="center">{user.email}</Text>
        </Flex>
      )}
      
      <Divider mb="6" />
      
      <VStack spacing="2">
        {sidebarItems.map((item) => (
          <SidebarItem
            key={item.path}
            icon={item.icon}
            label={item.label}
            path={item.path}
            active={pathname === item.path}
          />
        ))}
        
        <Flex
          align="center"
          p="4"
          mx="4"
          borderRadius="lg"
          role="group"
          cursor="pointer"
          onClick={handleLogout}
          color="gray.600"
          _hover={{
            bg: 'red.50',
            color: 'red.600',
          }}
        >
          <Icon mr="4" fontSize="16" as={FiLogOut} />
          <Text>Logout</Text>
        </Flex>
      </VStack>
    </Box>
  );

  return (
    <>
      {isDesktop ? (
        <SidebarContent />
      ) : (
        <Drawer
          autoFocus={false}
          isOpen={isOpen}
          placement="left"
          onClose={onClose}
          returnFocusOnClose={false}
          onOverlayClick={onClose}
        >
          <DrawerOverlay />
          <DrawerContent>
            <DrawerCloseButton />
            <DrawerHeader borderBottomWidth="1px">Teacher Portal</DrawerHeader>
            <DrawerBody p="0">
              <SidebarContent />
            </DrawerBody>
          </DrawerContent>
        </Drawer>
      )}
      
      <Box ml={{ base: 0, md: 60 }} p="4">
        {!isDesktop && (
          <HStack mb="4" justify="space-between">
            <IconButton
              aria-label="Open menu"
              icon={<FiMenu />}
              onClick={onOpen}
              variant="outline"
            />
            <Heading size="md" color="blue.600">Teacher Portal</Heading>
          </HStack>
        )}
        {children}
      </Box>
    </>
  );
}
