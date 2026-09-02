import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import VncView from '../views/VncView.vue'

const routes: Array<RouteRecordRaw> = [{ path: '/', name: 'vnc', component: VncView }]

export default createRouter({ history: createWebHistory(process.env.BASE_URL), routes })
